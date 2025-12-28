package com.github.dermatoai.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.github.dermatoai.repository.PredictionRepository
import com.github.dermatoai.screen.PredictionHistory
import com.github.dermatoai.state.HomeUiState
import com.github.dermatoai.state.NetworkProtocol
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch


class AnalyzeVM(application: Application) : AndroidViewModel(application) {
    private val _uiState = MutableStateFlow(HomeUiState())

    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    private val repository = PredictionRepository(application.applicationContext)

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
                val result = repository.predict(
                    uri = uri,
                    protocol = currentState.selectedProtocol
                )

                _uiState.update { state ->
                    val newHistory = listOf(result) + state.history
                    state.copy(
                        isLoading = false,
                        lastPredictionResult = result,
                        history = newHistory
                    )
                }
            } catch (e: Exception) {
                e.printStackTrace()
                _uiState.update {
                    it.copy(isLoading = false, errorMessage = e.message ?: "Unknown Error")
                }
            }
        }
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