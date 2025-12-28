package com.github.dermatoai.api

import com.github.dermatoai.api.rest.AnalyzeApiService
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object NetworkModule {
    private const val BASE_URL = "http://10.0.2.2:8080/"
    private const val GRPC_HOST = "10.0.2.2"
    private const val GRPC_PORT = 50051

    val restApi: AnalyzeApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(AnalyzeApiService::class.java)
    }

    val grpcChannel: ManagedChannel by lazy {
        ManagedChannelBuilder
            .forAddress(GRPC_HOST, GRPC_PORT)
            .usePlaintext()
            .build()
    }
}