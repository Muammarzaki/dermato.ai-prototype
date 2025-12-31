package com.github.dermatoai.ui.vm

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.github.dermatoai.data.repository.NetworkAnalyzeApiRepository
import com.github.dermatoai.ui.screen.PredictionHistory
import com.github.dermatoai.ui.state.HomeUiState
import com.github.dermatoai.domain.common.NetworkProtocol
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class AnalyzeVM @Inject constructor(private val repository: NetworkAnalyzeApiRepository) : ViewModel() {
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