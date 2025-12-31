package com.github.dermatoai.ui.vm

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.domain.common.Resource
import com.github.dermatoai.domain.entity.DiagnosisSession
import com.github.dermatoai.domain.usecase.AnalyzeUseCase
import com.github.dermatoai.domain.usecase.DataUseCase
import com.github.dermatoai.ui.screen.PredictionHistory
import com.github.dermatoai.ui.state.HomeUiState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class AnalyzeVM @Inject constructor(
    private val dataUseCase: DataUseCase,
    private val analyzeUseCase: AnalyzeUseCase
) : ViewModel() {

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

        viewModelScope.launch {
            analyzeUseCase(uri.toString(), currentState.selectedProtocol)
                .collect { resource ->
                    when (resource) {
                        is Resource.Loading -> {
                            _uiState.update { it.copy(isLoading = true, errorMessage = null) }
                        }

                        is Resource.Success -> {
                            val resultSession = resource.data

                            _uiState.update { state ->
                                val historyItem = mapDomainToUiHistory(resultSession)

                                val newHistory = listOf(historyItem) + state.history

                                state.copy(
                                    isLoading = false,
                                    lastPredictionResult = historyItem,
                                    history = newHistory
                                )
                            }
                        }

                        is Resource.Error -> {
                            _uiState.update {
                                it.copy(
                                    isLoading = false,
                                    errorMessage = resource.message
                                )
                            }
                        }
                    }
                }
        }
    }

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    private fun mapDomainToUiHistory(domain: DiagnosisSession): PredictionHistory {
        return PredictionHistory(
            id = domain.id,
            imageName = domain.image?.imageUri ?: "",
            result = domain.disease.name,
            confidence = "${(domain.disease.confidence * 100).toInt()}%",
            method = domain.metrics?.protocolUsed ?: "Unknown",
        )
    }

    private fun loadInitialHistory() {
        val dummy = listOf(
            PredictionHistory("1", "history_1.jpg", "Healthy", "99%", "gRPC"),
            PredictionHistory("2", "history_2.jpg", "Acne", "88%", "REST")
        )
        _uiState.update { it.copy(history = dummy) }
    }
}