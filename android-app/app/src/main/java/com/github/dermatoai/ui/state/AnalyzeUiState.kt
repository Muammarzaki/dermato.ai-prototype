package com.github.dermatoai.ui.state

import android.net.Uri
import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.ui.dto.PredictionHistory

data class AnalyzeUiState(
    val selectedProtocol: NetworkProtocol = NetworkProtocol.REST,
    val selectedImageUri: Uri? = null,
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val lastPredictionResult: PredictionHistory? = null
)