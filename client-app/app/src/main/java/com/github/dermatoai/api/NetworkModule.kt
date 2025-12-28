package com.github.dermatoai.api

import com.github.dermatoai.BuildConfig
import com.github.dermatoai.api.rest.AnalyzeApiService
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object NetworkModule {
    private const val BASE_URL = BuildConfig.BASE_URL
    private const val GRPC_HOST = BuildConfig.GRPC_HOST
    private val GRPC_PORT = BuildConfig.GRPC_PORT

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