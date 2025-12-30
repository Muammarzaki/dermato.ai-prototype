package com.github.dermatoai.domain.entity

data class DiagnosisSession(
    val id: String,
    val disease: DiseaseResult,
    val image: ImageInfo,
    val metrics: PerformanceMetrics,
    val timestamp: Long
)