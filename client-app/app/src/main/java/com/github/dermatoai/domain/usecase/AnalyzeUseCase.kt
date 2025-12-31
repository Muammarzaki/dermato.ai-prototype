package com.github.dermatoai.domain.usecase

import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.domain.common.Resource
import com.github.dermatoai.domain.entity.DiagnosisSession
import kotlinx.coroutines.flow.Flow

interface AnalyzeUseCase {
    operator fun invoke(
        imageUri: String,
        protocol: NetworkProtocol
    ): Flow<Resource<DiagnosisSession>>
}