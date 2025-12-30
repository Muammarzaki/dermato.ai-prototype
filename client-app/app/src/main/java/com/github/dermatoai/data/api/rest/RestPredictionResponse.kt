package com.github.dermatoai.data.api.rest

data class RestPredictionResponse(
    val class_name: String,
    val confidence: Double
)