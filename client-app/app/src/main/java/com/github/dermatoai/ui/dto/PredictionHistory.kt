package com.github.dermatoai.ui.dto

import com.github.dermatoai.domain.entity.DiagnosisSession

data class PredictionHistory(
    val id: String,
    val imageName: String,
    val result: String,
    val confidence: String,
    val method: String
) {
    companion object {
        fun mapDomain(domain: DiagnosisSession): PredictionHistory {
            return PredictionHistory(
                id = domain.id,
                imageName = domain.image?.imageUri ?: "",
                result = domain.disease.name,
                confidence = "${(domain.disease.confidence * 100).toInt()}%",
                method = domain.metrics?.protocolUsed ?: "Unknown",
            )
        }
    }
}