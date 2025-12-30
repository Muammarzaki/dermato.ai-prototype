package com.github.dermatoai.data.api.rest

import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface AnalyzeApiService {
    @Multipart
    @POST("/analyze-skin")
    suspend fun predictImage(
        @Part image: MultipartBody.Part,

        @Part("user_id") userId: RequestBody
    ): RestPredictionResponse
}