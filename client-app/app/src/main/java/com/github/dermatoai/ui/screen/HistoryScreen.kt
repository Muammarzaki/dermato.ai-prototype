package com.github.dermatoai.ui.screen

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.paging.LoadState
import androidx.paging.compose.collectAsLazyPagingItems
import androidx.paging.compose.itemContentType
import androidx.paging.compose.itemKey
import com.github.dermatoai.ui.components.HistoryItemCard
import com.github.dermatoai.ui.vm.DataVM

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(
    viewModel: DataVM = hiltViewModel()
) {
    // 1. Ambil Data Paging & State Filter
    val historyItems = viewModel.historyPagingFlow.collectAsLazyPagingItems()
    val filterState by viewModel.filterState.collectAsState()

    // State lokal untuk search bar agar responsif saat mengetik
    // Kita sinkronkan dengan filterState.label saat inisialisasi
    var searchText by remember(filterState.label) { mutableStateOf(filterState.label ?: "") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // --- SEARCH BAR ---
        OutlinedTextField(
            value = searchText,
            onValueChange = {
                searchText = it
                viewModel.updateFilter(query = it) // Update ViewModel
            },
            label = { Text("Search History (e.g. Melanoma)") },
            modifier = Modifier.fillMaxWidth(),
            leadingIcon = { Icon(Icons.Default.Search, null) },
            trailingIcon = if (searchText.isNotEmpty()) {
                {
                    IconButton(onClick = {
                        searchText = ""
                        viewModel.updateFilter(query = "")
                    }) {
                        Icon(Icons.Default.Close, contentDescription = "Clear")
                    }
                }
            } else null,
            singleLine = true
        )

        Spacer(modifier = Modifier.height(12.dp))

        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            item {
                FilterChip(
                    selected = filterState.protocol == null,
                    onClick = { viewModel.updateFilter(protocol = null) }, // Reset (All)
                    label = { Text("All") }
                )
            }
            item {
                FilterChip(
                    selected = filterState.protocol == "REST",
                    onClick = { viewModel.updateFilter(protocol = "REST") },
                    label = { Text("REST") }
                )
            }
            item {
                FilterChip(
                    selected = filterState.protocol == "GRPC", // Sesuaikan string dengan enum/data Anda (misal "GRPC" atau "gRPC")
                    onClick = { viewModel.updateFilter(protocol = "GRPC") },
                    label = { Text("gRPC") }
                )
            }

            item {
                Spacer(modifier = Modifier.width(4.dp))
                VerticalDivider(modifier = Modifier.height(32.dp))
                Spacer(modifier = Modifier.width(4.dp))
            }

            item {
                FilterChip(
                    selected = filterState.successOnly == true,
                    onClick = {
                        val newValue = if (filterState.successOnly == true) null else true
                        viewModel.updateFilter(successOnly = newValue)
                    },
                    label = { Text("Success Only") },
                    leadingIcon = if (filterState.successOnly == true) {
                        { Icon(Icons.Default.Check, null) }
                    } else null
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(8.dp),
            contentPadding = PaddingValues(bottom = 80.dp),
            modifier = Modifier.weight(1f) // Isi sisa layar
        ) {
            if (historyItems.loadState.refresh is LoadState.Loading) {
                item {
                    Box(
                        modifier = Modifier.fillMaxWidth().padding(32.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator()
                    }
                }
            }

            if (historyItems.loadState.refresh is LoadState.NotLoading && historyItems.itemCount == 0) {
                item {
                    Box(
                        modifier = Modifier.fillMaxWidth().padding(32.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("No history found.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.secondary)
                    }
                }
            }

            items(
                count = historyItems.itemCount,
                key = historyItems.itemKey { it.id },
                contentType = historyItems.itemContentType { "history_item" }
            ) { index ->
                val item = historyItems[index]
                if (item != null) {
                    HistoryItemCard(item = item)
                }
            }

            item {
                if (historyItems.loadState.append is LoadState.Loading) {
                    Box(
                        modifier = Modifier.fillMaxWidth().padding(8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(24.dp))
                    }
                }
            }
        }
    }
}