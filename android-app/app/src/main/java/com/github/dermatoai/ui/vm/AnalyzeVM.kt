package com.github.dermatoai.ui.vm

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.domain.common.Resource
import com.github.dermatoai.domain.usecase.AnalyzeUseCase
import com.github.dermatoai.ui.dto.PredictionHistory
import com.github.dermatoai.ui.state.AnalyzeUiState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class AnalyzeVM @Inject constructor(
    private val analyzeUseCase: AnalyzeUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow(AnalyzeUiState())
    val uiState: StateFlow<AnalyzeUiState> = _uiState.asStateFlow()

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
                                val historyItem = PredictionHistory.mapDomain(resultSession)

                                state.copy(
                                    isLoading = false,
                                    lastPredictionResult = historyItem,
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

    fun resetUIState() {
        _uiState.value = _uiState.value.copy(
            selectedImageUri = null,
            lastPredictionResult = null,
            errorMessage = null,
            isLoading = false,
        )
    }

}