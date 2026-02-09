package com.github.dermatoai.domain.usecase

import com.github.dermatoai.data.db.dto.ProtocolStatDto
import com.github.dermatoai.domain.repository.LocalDBRepository
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

class StatisticUseCaseImpl @Inject constructor(
    private val repository: LocalDBRepository
) : StatisticUseCase {
    override fun getProtocolStats(): Flow<List<ProtocolStatDto>> {
        return repository.getProtocolStats()
    }

    override fun getTotalRecord(): Flow<Int> {
        return repository.getRecordCount()
    }
}