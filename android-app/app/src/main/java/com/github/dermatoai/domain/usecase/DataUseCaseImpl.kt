package com.github.dermatoai.domain.usecase

import androidx.paging.PagingData
import com.github.dermatoai.domain.entity.DiagnosisSession
import com.github.dermatoai.domain.entity.PredictionFilter
import com.github.dermatoai.domain.repository.LocalDBRepository
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

class DataUseCaseImpl @Inject constructor(
    private val repository: LocalDBRepository
) : DataUseCase {
    override suspend fun savePrediction(session: DiagnosisSession): Long =
        repository.savePrediction(session)

    override suspend fun deletePrediction(id: Long) = repository.deletePrediction(id)

    override suspend fun getPredictionById(id: Long): DiagnosisSession? =
        repository.getPredictionById(id)

    override fun getPredictionHistory(
        filter: PredictionFilter
    ): Flow<PagingData<DiagnosisSession>> = repository.getPredictionHistory(filter)
}