package com.github.dermatoai.domain.entity

data class PredictionFilter(
    val protocol: String? = null,
    val successOnly: Boolean? = null,
    val label: String? = null,
    val fromDate: Long? = null,
    val toDate: Long? = null
)