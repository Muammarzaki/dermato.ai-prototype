package com.github.dermatoai.data.api.dto

import com.github.dermatoai.data.api.dto.AnalyzeResultItemDTO
import com.google.gson.annotations.SerializedName

data class AnalyzeApiResponseDTO(
    @SerializedName("analysis_id")
    val analysisId: String,

    @SerializedName("analysis_timestamp")
    val analysisTimestamp: String,

    @SerializedName("results")
    val results: List<AnalyzeResultItemDTO>
)