package com.github.dermatoai.state

import android.net.Uri
import com.github.dermatoai.screen.PredictionHistory

data class HomeUiState(
    val selectedProtocol: NetworkProtocol = NetworkProtocol.REST,
    val selectedImageUri: Uri? = null,
    val history: List<PredictionHistory> = emptyList(),
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val lastPredictionResult: PredictionHistory? = null
)