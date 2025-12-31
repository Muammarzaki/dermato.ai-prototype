package com.github.dermatoai.ui.state

import android.net.Uri
import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.ui.screen.PredictionHistory

data class HomeUiState(
    val selectedProtocol: NetworkProtocol = NetworkProtocol.REST,
    val selectedImageUri: Uri? = null,
    val history: List<PredictionHistory> = emptyList(),
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val lastPredictionResult: PredictionHistory? = null
)