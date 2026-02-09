package com.github.dermatoai.ui.state

import com.github.dermatoai.ui.dto.ProtocolUiData

data class StatisticsUiState(
    val totalScans: Int = 0,
    val overallSuccessRate: Float = 0f,

    val restStats: ProtocolUiData = ProtocolUiData("REST"),
    val grpcStats: ProtocolUiData = ProtocolUiData("GRPC"),

    val isLoading: Boolean = true
)