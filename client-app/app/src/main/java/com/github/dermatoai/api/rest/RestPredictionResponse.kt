package com.github.dermatoai.api.rest

data class RestPredictionResponse(
    val class_name: String,
    val confidence: Double
)