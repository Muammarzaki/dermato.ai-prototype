package com.github.dermatoai.ui.dto

import android.text.format.DateUtils
import com.github.dermatoai.domain.entity.DiagnosisSession

data class PredictionHistory(
    val id: String,
    val rawId: Long,
    val imagePath: String,
    val prediction: String,
    val confidence: Double,
    val method: String,
    val latency: Long,
    val timestamp: Long,
    val isSuccess: Boolean
) {
    fun getRelativeTime(): String {
        return DateUtils.getRelativeTimeSpanString(
            timestamp,
            System.currentTimeMillis(),
            DateUtils.MINUTE_IN_MILLIS
        ).toString()
    }

    fun getFormattedLatency(): String = "${latency}ms"

    fun getFormattedConfidence(): String = "${(confidence * 100).toInt()}%"

    companion object {
        fun mapDomain(domain: DiagnosisSession): PredictionHistory {
            return PredictionHistory(
                id = domain.id,
                rawId = domain.id.toLongOrNull() ?: 0L,
                imagePath = domain.image?.imageUri ?: "",
                prediction = domain.disease.name,
                confidence = domain.disease.confidence.toDouble(),
                method = domain.metrics?.protocolUsed ?: "Unknown",
                latency = domain.metrics?.latencyMs ?: 0L,
                timestamp = domain.timestamp,
                isSuccess = domain.metrics?.status
                    ?: false
            )
        }
    }
}