package com.github.dermatoai.domain.usecase

import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.android.components.ViewModelComponent

@Module
@InstallIn(ViewModelComponent::class)
abstract class UseCaseModule {
    @Binds
    abstract fun bindAnalyzeUseCase(
        impl: AnalyzeUseCaseImpl
    ): AnalyzeUseCase

    @Binds
    abstract fun bindDataUseCase(
        impl: DataUseCaseImpl
    ): DataUseCase
}