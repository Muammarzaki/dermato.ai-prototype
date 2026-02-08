package com.github.dermatoai.ui.vm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.github.dermatoai.data.db.dto.ProtocolStatDto
import com.github.dermatoai.domain.repository.LocalDBRepository
import com.github.dermatoai.ui.dto.ProtocolUiData
import com.github.dermatoai.ui.state.StatisticsUiState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import javax.inject.Inject

@HiltViewModel
class StatisticsVM @Inject constructor(
    repository: LocalDBRepository
) : ViewModel() {

    val uiState: StateFlow<StatisticsUiState> = repository.getProtocolStats()
        .map { rawList ->
            calculateStats(rawList)
        }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = StatisticsUiState(isLoading = true)
        )

    private fun calculateStats(list: List<ProtocolStatDto>): StatisticsUiState {
        val restDto = list.find { it.protocol.equals("REST", ignoreCase = true) }
        val grpcDto = list.find { it.protocol.equals("GRPC", ignoreCase = true) }

        val restTotal = restDto?.totalCount ?: 0
        val grpcTotal = grpcDto?.totalCount ?: 0
        val grandTotal = restTotal + grpcTotal

        if (grandTotal == 0) {
            return StatisticsUiState(isLoading = false)
        }

        fun mapToUiData(dto: ProtocolStatDto?, protocolName: String): ProtocolUiData {
            if (dto == null || dto.totalCount == 0) return ProtocolUiData(protocolName)

            val successRate = dto.successCount.toFloat() / dto.totalCount
            
            return ProtocolUiData(
                name = protocolName,
                avgLatencyMs = dto.avgLatencyMs,
                totalCount = dto.totalCount,
                usagePercent = dto.totalCount.toFloat() / grandTotal,
                successRate = successRate * 100f,
                errorRate = (1f - successRate) * 100f
            )
        }

        val restUiData = mapToUiData(restDto, "REST")
        val grpcUiData = mapToUiData(grpcDto, "gRPC")

        val totalSuccess = (restDto?.successCount ?: 0) + (grpcDto?.successCount ?: 0)
        val overallSuccess = if (grandTotal > 0) (totalSuccess.toFloat() / grandTotal) * 100f else 0f

        return StatisticsUiState(
            totalScans = grandTotal,
            overallSuccessRate = overallSuccess,
            restStats = restUiData,
            grpcStats = grpcUiData,
            isLoading = false
        )
    }
}