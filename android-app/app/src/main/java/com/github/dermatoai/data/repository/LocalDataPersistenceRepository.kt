package com.github.dermatoai.data.repository

import androidx.paging.Pager
import androidx.paging.PagingConfig
import androidx.paging.PagingData
import androidx.paging.map
import com.github.dermatoai.data.db.dao.PredictionRecordDao
import com.github.dermatoai.data.db.dto.ProtocolStatDto
import com.github.dermatoai.data.db.utils.PredictionQueryBuilder
import com.github.dermatoai.data.mapper.toDomain
import com.github.dermatoai.data.mapper.toEntity
import com.github.dermatoai.domain.entity.DiagnosisSession
import com.github.dermatoai.domain.entity.PredictionFilter
import com.github.dermatoai.domain.repository.LocalDBRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject


class LocalDataPersistenceRepository @Inject constructor(
    private val dao: PredictionRecordDao
) : LocalDBRepository {

    override suspend fun savePrediction(session: DiagnosisSession): Long {
        return dao.insert(session.toEntity())
    }

    override suspend fun deletePrediction(id: Long) {
        dao.deleteById(id)
    }

    override suspend fun getPredictionById(id: Long): DiagnosisSession? {
        val entity = dao.getById(id)
        return entity?.toDomain()
    }

    override fun getPredictionHistory(
        filter: PredictionFilter
    ): Flow<PagingData<DiagnosisSession>> {

        return Pager(
            config = PagingConfig(pageSize = 20),
            pagingSourceFactory = {
                dao.pagingWithFilter(PredictionQueryBuilder.build(filter))
            }
        ).flow
            .map { pagingData ->
                pagingData.map { entity ->
                    entity.toDomain()
                }
            }
    }

    override fun getProtocolStats(): Flow<List<ProtocolStatDto>> {
        return dao.getProtocolStatistics()
    }

    override fun deleteAllPrediction(): Boolean {
        dao.deleteAll()
        return true
    }
}