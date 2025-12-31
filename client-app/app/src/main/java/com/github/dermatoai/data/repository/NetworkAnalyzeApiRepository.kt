package com.github.dermatoai.data.repository

import android.content.Context
import android.net.Uri
import com.github.dermatoai.AnalyzeSkinRequest
import com.github.dermatoai.AnalyzeSkinResponse
import com.github.dermatoai.ImageInfo
import com.github.dermatoai.SkinAnalysisServiceGrpcKt
import com.github.dermatoai.data.api.dto.AnalyzeApiResponseDTO
import com.github.dermatoai.data.api.rest.AnalyzeApiService
import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.domain.entity.DiagnosisSession
import com.github.dermatoai.domain.entity.DiseaseResult
import com.github.dermatoai.domain.entity.PerformanceMetrics
import com.github.dermatoai.domain.repository.NetworkAnalyzeRepository
import com.google.protobuf.ByteString
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.time.Instant
import javax.inject.Inject
import kotlin.math.min
import kotlin.system.measureTimeMillis

class NetworkAnalyzeApiRepository @Inject constructor(
    private val analyzeApi: AnalyzeApiService,
    private val analyseStub: SkinAnalysisServiceGrpcKt.SkinAnalysisServiceCoroutineStub,
    @param:ApplicationContext private val context: Context
) : NetworkAnalyzeRepository {

    override suspend fun predict(
        uri: Uri,
        protocol: NetworkProtocol
    ): DiagnosisSession = withContext(Dispatchers.IO) {

        val imageBytes = readBytesFromUri(uri)
        var rawRestResponse: AnalyzeApiResponseDTO? = null
        var rawGrpcResponse: AnalyzeSkinResponse? = null

        val latency = measureTimeMillis {
            when (protocol) {
                NetworkProtocol.REST -> {
                    rawRestResponse = fetchRestRaw(imageBytes)
                }

                NetworkProtocol.GRPC -> {
                    rawGrpcResponse = fetchGrpcRaw(imageBytes)
                }
            }
        }

        if (protocol == NetworkProtocol.REST && rawRestResponse != null) {
            return@withContext mapRestToDomain(rawRestResponse, uri, latency)
        } else if (protocol == NetworkProtocol.GRPC && rawGrpcResponse != null) {
            return@withContext mapGrpcToDomain(rawGrpcResponse, uri, latency)
        } else {
            throw IllegalStateException("Response is null but no error was thrown")
        }
    }


    private suspend fun fetchRestRaw(imageBytes: ByteArray): AnalyzeApiResponseDTO {
        val requestBody = imageBytes.toRequestBody("image/jpeg".toMediaType())
        val imagePart = MultipartBody.Part.createFormData("file", "upload.jpg", requestBody)
        val userIdBody = "android-user".toRequestBody("text/plain".toMediaType())

        return analyzeApi.predictImage(imagePart, userIdBody)
    }

    private suspend fun fetchGrpcRaw(imageBytes: ByteArray): AnalyzeSkinResponse {
        val requestFlow = flow {
            emit(
                AnalyzeSkinRequest.newBuilder()
                    .setInfo(
                        ImageInfo.newBuilder().setImageType("jpeg").setUserId("android-user")
                            .build()
                    )
                    .build()
            )
            val chunkSize = 64 * 1024
            var offset = 0
            while (offset < imageBytes.size) {
                val length = min(chunkSize, imageBytes.size - offset)
                emit(
                    AnalyzeSkinRequest.newBuilder()
                        .setChunk(ByteString.copyFrom(imageBytes, offset, length)).build()
                )
                offset += length
            }
        }
        return analyseStub.analyzeSkin(requestFlow)
    }

    private fun mapRestToDomain(
        dto: AnalyzeApiResponseDTO,
        uri: Uri,
        latency: Long
    ): DiagnosisSession {
        val timeEpoch = try {
            Instant.parse(dto.analysisTimestamp).toEpochMilli()
        } catch (_: Exception) {
            System.currentTimeMillis()
        }

        return DiagnosisSession(
            id = dto.analysisId,
            disease = DiseaseResult(
                name = dto.results.firstOrNull()?.className ?: "Unknown",
                confidence = dto.results.firstOrNull()?.confidence ?: 0f,
            ),
            image = com.github.dermatoai.domain.entity.ImageInfo(
                imageUri = uri.toString(),
                imageWidth = null,
                imageHeight = null
            ),
            metrics = PerformanceMetrics(
                latencyMs = latency,
                protocolUsed = "REST",
                status = true,
            ),
            timestamp = timeEpoch
        )
    }

    private fun mapGrpcToDomain(
        proto: AnalyzeSkinResponse,
        uri: Uri,
        latency: Long
    ): DiagnosisSession {
        return DiagnosisSession(
            id = proto.analysisId,
            disease = DiseaseResult(
                name = proto.resultsList.firstOrNull()?.label ?: "Unknown",
                confidence = proto.resultsList.firstOrNull()?.confidence ?: 0f,
            ),
            image = com.github.dermatoai.domain.entity.ImageInfo(
                imageUri = uri.toString(),
                imageWidth = null,
                imageHeight = null
            ),
            metrics = PerformanceMetrics(
                latencyMs = latency,
                protocolUsed = "GRPC",
                status = true,
            ),
            timestamp = proto.analysisTimestamp.seconds * 1000
        )
    }

    private fun readBytesFromUri(uri: Uri): ByteArray {
        return context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            ?: throw IllegalArgumentException("Unable to read URI: $uri")
    }

}