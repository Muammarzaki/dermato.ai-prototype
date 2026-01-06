package com.github.dermatoai.data.api.dto

import com.google.gson.annotations.SerializedName

data class AnalyzeResultItemDTO(
    @SerializedName("label")
    val className: String,

    @SerializedName("confidence")
    val confidence: Float,

    @SerializedName("description")
    val description: String? = null,

    @SerializedName("recommendation")
    val recommendation: String? = null
)