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
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object APIModule {

    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        val loggingInterceptor = HttpLoggingInterceptor().apply {
            if (BuildConfig.BUILD_TYPE == "debug")
                level = HttpLoggingInterceptor.Level.BODY
        }

        return OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
    }

    @Provides
    @Singleton
    fun provideRetrofit(okHttpClient: OkHttpClient): Retrofit =
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
            .also {
                if (BuildConfig.BUILD_TYPE == "debug") {
                    it.usePlaintext()
                } else {
                    it.useTransportSecurity()
                }
            }
            .keepAliveTime(120, TimeUnit.SECONDS)
            .keepAliveTimeout(5, TimeUnit.SECONDS)
            .keepAliveWithoutCalls(true)
            .build()

    @Provides
    @Singleton
    fun provideSkinAnalysisStub(
        channel: ManagedChannel
    ): SkinAnalysisServiceGrpcKt.SkinAnalysisServiceCoroutineStub =
        SkinAnalysisServiceGrpcKt.SkinAnalysisServiceCoroutineStub(channel)
}