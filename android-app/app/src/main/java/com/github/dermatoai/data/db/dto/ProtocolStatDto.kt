package com.github.dermatoai.data.db.dto

import androidx.room.ColumnInfo

data class ProtocolStatDto(
    @ColumnInfo(name = "protocol") val protocol: String,
    @ColumnInfo(name = "total_count") val totalCount: Int,
    @ColumnInfo(name = "success_count") val successCount: Int,
    @ColumnInfo(name = "avg_latency_ms") val avgLatencyMs: Double
)