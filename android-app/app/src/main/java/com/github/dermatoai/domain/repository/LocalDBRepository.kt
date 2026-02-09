package com.github.dermatoai.domain.repository

import androidx.paging.PagingData
import com.github.dermatoai.data.db.dto.ProtocolStatDto
import com.github.dermatoai.domain.entity.DiagnosisSession
import com.github.dermatoai.domain.entity.PredictionFilter
import kotlinx.coroutines.flow.Flow

interface LocalDBRepository {
    suspend fun savePrediction(session: DiagnosisSession): Long

    suspend fun deletePrediction(id: Long)

    suspend fun getPredictionById(id: Long): DiagnosisSession?

    fun getPredictionHistory(
        filter: PredictionFilter
    ): Flow<PagingData<DiagnosisSession>>

    fun getProtocolStats(): Flow<List<ProtocolStatDto>>

    fun deleteAllPrediction(): Boolean

    fun getRecordCount(): Flow<Int>
}