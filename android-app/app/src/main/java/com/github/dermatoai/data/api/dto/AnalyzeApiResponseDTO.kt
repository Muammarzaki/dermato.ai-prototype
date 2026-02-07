package com.github.dermatoai.data.api.dto

import com.google.gson.annotations.SerializedName

data class AnalyzeApiResponseDTO(
    @SerializedName("analysis_id")
    val analysisId: String,

    @SerializedName("analysis_timestamp")
    val analysisTimestamp: String,

    @SerializedName("server_sha256")
    val serverSha256: String,

    @SerializedName("results")
    val results: List<AnalyzeResultItemDTO>
)