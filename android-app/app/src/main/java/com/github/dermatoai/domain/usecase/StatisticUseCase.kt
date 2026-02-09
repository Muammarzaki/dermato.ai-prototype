package com.github.dermatoai.domain.usecase

import com.github.dermatoai.data.db.dto.ProtocolStatDto
import kotlinx.coroutines.flow.Flow

interface StatisticUseCase {
    fun getProtocolStats(): Flow<List<ProtocolStatDto>>
    fun getTotalRecord(): Flow<Int>
}