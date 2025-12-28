package com.github.dermatoai.repository

import android.content.Context
import android.net.Uri
import com.github.dermatoai.AnalyzeSkinRequest
import com.github.dermatoai.ImageInfo
import com.github.dermatoai.SkinAnalysisServiceGrpcKt
import com.github.dermatoai.api.NetworkModule
import com.github.dermatoai.screen.PredictionHistory
import com.github.dermatoai.state.NetworkProtocol
import com.google.protobuf.ByteString
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream


class PredictionRepository(private val context: Context) {

    private suspend fun readBytesFromUri(uri: Uri): ByteArray = withContext(Dispatchers.IO) {
        val inputStream = context.contentResolver.openInputStream(uri)
        val byteBuffer = ByteArrayOutputStream()
        val bufferSize = 1024
        val buffer = ByteArray(bufferSize)
        var len = 0
        while (inputStream!!.read(buffer).also { len = it } != -1) {
            byteBuffer.write(buffer, 0, len)
        }
        inputStream.close()
        return@withContext byteBuffer.toByteArray()
    }

    suspend fun predict(
        uri: Uri,
        protocol: NetworkProtocol
    ): PredictionHistory = withContext(Dispatchers.IO) {

        val imageBytes = readBytesFromUri(uri)

        return@withContext when (protocol) {
            NetworkProtocol.REST -> fetchViaRest(imageBytes)
            NetworkProtocol.GRPC -> fetchViaGrpc(imageBytes)
        }
    }

    private suspend fun fetchViaRest(imageBytes: ByteArray): PredictionHistory {
        val requestFile = imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
        val bodyImage = MultipartBody.Part.createFormData("file", "upload.jpg", requestFile)
        val userIdBody = "user-android-client".toRequestBody("text/plain".toMediaTypeOrNull())

        val response = NetworkModule.restApi.predictImage(bodyImage, userIdBody)

        return PredictionHistory(
            id = (System.currentTimeMillis() % 10000).toInt(),
            imageName = "upload_rest.jpg",
            result = response.class_name,
            confidence = "${(response.confidence * 100).toInt()}%",
            method = "REST"
        )
    }

    private suspend fun fetchViaGrpc(imageBytes: ByteArray): PredictionHistory {

        val stub =
            SkinAnalysisServiceGrpcKt.SkinAnalysisServiceCoroutineStub(NetworkModule.grpcChannel)

        val requestFlow = flow {

            val info = ImageInfo.newBuilder()
                .setImageType("jpeg")
                .setUserId("android-user")
                .build()

            emit(
                AnalyzeSkinRequest.newBuilder()
                    .setInfo(info)
                    .build()
            )

            val chunkSize = 64 * 1024
            var offset = 0

            while (offset < imageBytes.size) {
                val length = kotlin.math.min(chunkSize, imageBytes.size - offset)

                val chunkByteString = ByteString.copyFrom(imageBytes, offset, length)

                emit(
                    AnalyzeSkinRequest.newBuilder()
                        .setChunk(chunkByteString)
                        .build()
                )

                offset += length
            }
        }

        try {
            val response = stub.analyzeSkin(requestFlow)

            val topResult = response.resultsList.firstOrNull()

            return PredictionHistory(
                id = (System.currentTimeMillis() % 10000).toInt(),
                imageName = "upload_grpc.jpg",
                result = topResult?.label ?: "Unknown",
                confidence = topResult?.let { "${(it.confidence * 100).toInt()}%" } ?: "0%",
                method = "gRPC"
            )

        } catch (e: Exception) {
            e.printStackTrace()
            return PredictionHistory(
                id = 0,
                imageName = "error.jpg",
                result = "Error: ${e.message}",
                confidence = "0%",
                method = "gRPC Failed"
            )
        }
    }
}