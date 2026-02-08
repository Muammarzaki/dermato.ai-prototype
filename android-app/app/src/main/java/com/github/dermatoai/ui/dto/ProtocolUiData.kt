package com.github.dermatoai.ui.dto

data class ProtocolUiData(
    val name: String,
    val avgLatencyMs: Double = 0.0,
    val usagePercent: Float = 0f, // 0.0 - 1.0
    val successRate: Float = 0f,  // 0.0 - 1.0
    val errorRate: Float = 0f,    // 0.0 - 1.0
    val totalCount: Int = 0
)