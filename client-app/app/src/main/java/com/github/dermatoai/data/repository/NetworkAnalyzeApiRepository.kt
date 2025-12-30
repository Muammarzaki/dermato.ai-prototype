package com.github.dermatoai.data.repository

import android.content.Context
import android.net.Uri
import com.github.dermatoai.AnalyzeSkinRequest
import com.github.dermatoai.ImageInfo
import com.github.dermatoai.SkinAnalysisServiceGrpcKt
import com.github.dermatoai.data.api.rest.AnalyzeApiService
import com.github.dermatoai.domain.enum.NetworkProtocol
import com.github.dermatoai.domain.repository.NetworkAnalyzeRepository
import com.github.dermatoai.ui.screen.PredictionHistory
import com.google.protobuf.ByteString
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import javax.inject.Inject
import kotlin.math.min


class NetworkAnalyzeApiRepository @Inject constructor(
    private val analyzeApi: AnalyzeApiService,
    private val analyseStub: SkinAnalysisServiceGrpcKt.SkinAnalysisServiceCoroutineStub,
    @param:ApplicationContext private val context: Context
) : NetworkAnalyzeRepository {

    override suspend fun predict(
        uri: Uri,
        protocol: NetworkProtocol
    ): PredictionHistory = withContext(Dispatchers.IO) {

        val imageBytes = readBytesFromUri(uri)

        when (protocol) {
            NetworkProtocol.REST -> fetchViaRest(imageBytes)
            NetworkProtocol.GRPC -> fetchViaGrpc(imageBytes)
        }
    }

    private suspend fun fetchViaRest(imageBytes: ByteArray): PredictionHistory {

        val requestBody = imageBytes.toRequestBody("image/jpeg".toMediaType())
        val imagePart = MultipartBody.Part.createFormData(
            name = "file",
            filename = "upload.jpg",
            body = requestBody
        )

        val userIdBody =
            "android-user".toRequestBody("text/plain".toMediaType())

        val response = analyzeApi.predictImage(imagePart, userIdBody)

        return PredictionHistory(
            id = generateId(),
            imageName = "upload_rest.jpg",
            result = response.class_name,
            confidence = "${(response.confidence * 100).toInt()}%",
            method = "REST"
        )
    }

    private suspend fun fetchViaGrpc(imageBytes: ByteArray): PredictionHistory {

        val requestFlow = flow {

            emit(
                AnalyzeSkinRequest.newBuilder()
                    .setInfo(
                        ImageInfo.newBuilder()
                            .setImageType("jpeg")
                            .setUserId("android-user")
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
                        .setChunk(
                            ByteString.copyFrom(imageBytes, offset, length)
                        )
                        .build()
                )

                offset += length
            }
        }

        val response = analyseStub.analyzeSkin(requestFlow)
        val topResult = response.resultsList.firstOrNull()

        return PredictionHistory(
            id = generateId(),
            imageName = "upload_grpc.jpg",
            result = topResult?.label ?: "Unknown",
            confidence = topResult?.let { "${(it.confidence * 100).toInt()}%" } ?: "0%",
            method = "gRPC"
        )
    }

    private fun generateId(): Int =
        (System.currentTimeMillis() % Int.MAX_VALUE).toInt()

    private fun readBytesFromUri(uri: Uri): ByteArray =
        context.contentResolver.openInputStream(uri)?.use { inputStream ->

            val outputStream = ByteArrayOutputStream()
            val buffer = ByteArray(1024)
            var len: Int

            while (inputStream.read(buffer).also { len = it } != -1) {
                outputStream.write(buffer, 0, len)
            }

            outputStream.toByteArray()

        } ?: throw IllegalArgumentException("Unable to open InputStream for URI: $uri")
}