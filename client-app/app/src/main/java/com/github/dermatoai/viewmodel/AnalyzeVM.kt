package com.github.dermatoai.viewmodel

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.github.dermatoai.screen.PredictionHistory
import com.github.dermatoai.state.HomeUiState
import com.github.dermatoai.state.NetworkProtocol
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch


class AnalyzeVM : ViewModel() {
    private val _uiState = MutableStateFlow(HomeUiState())

    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    init {
        loadInitialHistory()
    }

    fun onProtocolSelected(protocol: NetworkProtocol) {
        _uiState.update { it.copy(selectedProtocol = protocol) }
    }

    fun onImageSelected(uri: Uri?) {
        _uiState.update { it.copy(selectedImageUri = uri, errorMessage = null) }
    }

    fun analyzeImage() {
        val currentState = _uiState.value
        val uri = currentState.selectedImageUri

        if (uri == null) {
            _uiState.update { it.copy(errorMessage = "Please select an image first.") }
            return
        }

        _uiState.update { it.copy(isLoading = true, errorMessage = null) }

        viewModelScope.launch {
            try {
                val result = when (currentState.selectedProtocol) {
                    NetworkProtocol.REST -> performRestPrediction(uri)
                    NetworkProtocol.GRPC -> performGrpcPrediction(uri)
                }

                _uiState.update { state ->
                    val newHistory = listOf(result) + state.history
                    state.copy(
                        isLoading = false,
                        lastPredictionResult = result,
                        history = newHistory
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(isLoading = false, errorMessage = e.message) }
            }
        }
    }

    private suspend fun performRestPrediction(uri: Uri): PredictionHistory {
        return PredictionHistory(
            id = (System.currentTimeMillis() % 1000).toInt(),
            imageName = "img_rest.jpg",
            result = "Eczema (REST)",
            confidence = "92.5%",
            method = "REST"
        )
    }

    private suspend fun performGrpcPrediction(uri: Uri): PredictionHistory {
        delay(1000)
        return PredictionHistory(
            id = (System.currentTimeMillis() % 1000).toInt(),
            imageName = "img_grpc.jpg",
            result = "Melanoma (gRPC)",
            confidence = "98.1%",
            method = "gRPC"
        )
    }

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    private fun loadInitialHistory() {
        val dummy = listOf(
            PredictionHistory(1, "history_1.jpg", "Healthy", "99%", "gRPC"),
            PredictionHistory(2, "history_2.jpg", "Acne", "88%", "REST")
        )
        _uiState.update { it.copy(history = dummy) }
    }
}