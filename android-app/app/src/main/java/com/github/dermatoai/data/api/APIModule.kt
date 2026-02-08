package com.github.dermatoai.data.api

import com.github.dermatoai.BuildConfig
import com.github.dermatoai.SkinAnalysisServiceGrpcKt
import com.github.dermatoai.data.api.rest.AnalyzeApiService
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object APIModule {

    val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .build()

    @Provides
    @Singleton
    fun provideRetrofit(): Retrofit =
        Retrofit.Builder()
            .baseUrl(BuildConfig.BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

    @Provides
    @Singleton
    fun provideAnalyzeApiService(
        retrofit: Retrofit
    ): AnalyzeApiService =
        retrofit.create(AnalyzeApiService::class.java)

    @Provides
    @Singleton
    fun provideGrpcChannel(): ManagedChannel =
        ManagedChannelBuilder
            .forAddress(BuildConfig.GRPC_HOST, BuildConfig.GRPC_PORT)
            .usePlaintext()
            .build()

    @Provides
    @Singleton
    fun provideSkinAnalysisStub(
        channel: ManagedChannel
    ): SkinAnalysisServiceGrpcKt.SkinAnalysisServiceCoroutineStub =
        SkinAnalysisServiceGrpcKt.SkinAnalysisServiceCoroutineStub(channel)
}