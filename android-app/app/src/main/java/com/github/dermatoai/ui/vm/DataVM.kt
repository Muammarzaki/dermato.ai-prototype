package com.github.dermatoai.ui.vm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.paging.PagingData
import androidx.paging.cachedIn
import androidx.paging.map
import com.github.dermatoai.domain.entity.PredictionFilter
import com.github.dermatoai.domain.usecase.DataUseCase
import com.github.dermatoai.ui.dto.PredictionHistory
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

@HiltViewModel
class DataVM @Inject constructor(
    private val dataUseCase: DataUseCase
) : ViewModel() {

    private val _filterState = MutableStateFlow(PredictionFilter())
    val filterState = _filterState.asStateFlow()

    @OptIn(ExperimentalCoroutinesApi::class)
    val historyPagingFlow: Flow<PagingData<PredictionHistory>> = _filterState
        .flatMapLatest { currentFilter ->
            dataUseCase.getPredictionHistory(currentFilter)
        }
        .map { pagingData ->
            pagingData.map { session -> PredictionHistory.mapDomain(session) }
        }
        .cachedIn(viewModelScope)

    fun updateFilter(
        query: String? = _filterState.value.label,
        successOnly: Boolean? = _filterState.value.successOnly,
        protocol: String? = _filterState.value.protocol
    ) {
        val current = _filterState.value
        _filterState.value = current.copy(
            label = query,
            successOnly = successOnly,
            protocol = protocol
        )
    }

    fun deletePrediction(id: Long) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                dataUseCase.deletePrediction(id)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    fun deleteAllHistory() {
        try {
            viewModelScope.launch(Dispatchers.IO) {
                dataUseCase.deleteAllPrediction().let {
                    withContext(Dispatchers.Main) {
                        if (it) {
                            _filterState.value = PredictionFilter()
                        }
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}