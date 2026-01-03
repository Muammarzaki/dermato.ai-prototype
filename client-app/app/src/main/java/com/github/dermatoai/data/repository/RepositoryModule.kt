package com.github.dermatoai.data.repository

import com.github.dermatoai.domain.repository.LocalDBRepository
import com.github.dermatoai.domain.repository.NetworkAnalyzeRepository
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {
    @Binds
    @Singleton
    abstract fun bindLocalDBRepository(
        impl: LocalDataPersistenceRepository
    ): LocalDBRepository

    @Binds
    @Singleton
    abstract fun bindNetworkAnalyzeApiRepository(
        impl: NetworkAnalyzeApiRepository
    ): NetworkAnalyzeRepository

}