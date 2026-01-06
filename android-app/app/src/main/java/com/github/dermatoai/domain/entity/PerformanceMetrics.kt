package com.github.dermatoai.domain.entity

data class PerformanceMetrics(
    val latencyMs: Long,
    val protocolUsed: String,
    val status: Boolean
)