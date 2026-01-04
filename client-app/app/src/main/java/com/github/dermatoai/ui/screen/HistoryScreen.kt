package com.github.dermatoai.ui.screen

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.paging.LoadState
import androidx.paging.compose.collectAsLazyPagingItems
import androidx.paging.compose.itemKey
import com.github.dermatoai.ui.components.HistoryItemCard
import com.github.dermatoai.ui.theme.DermatoaiTheme
import com.github.dermatoai.ui.vm.DataVM

@Composable
fun HistoryScreen(
    viewModel: DataVM = hiltViewModel()
) {
    val historyItems = viewModel.historyPagingFlow.collectAsLazyPagingItems()

    var searchText by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        OutlinedTextField(
            value = searchText,
            onValueChange = {
                searchText = it
                viewModel.updateFilter(query = it) // Realtime search
            },
            label = { Text("Search History (e.g. Melanoma)") },
            modifier = Modifier.fillMaxWidth(),
            leadingIcon = { Icon(Icons.Default.Search, null) }
        )

        Spacer(modifier = Modifier.height(16.dp))

        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(8.dp),
            contentPadding = PaddingValues(bottom = 80.dp)
        ) {
            items(
                count = historyItems.itemCount,
                key = historyItems.itemKey { it.id }
            ) { index ->
                val item = historyItems[index]
                if (item != null) {
                    // Reuse Component Card Anda yang sudah ada
                    HistoryItemCard(item = item)
                }
            }

            item {
                if (historyItems.loadState.append is LoadState.Loading) {
                    CircularProgressIndicator(modifier = Modifier.align(Alignment.CenterHorizontally))
                }
            }
        }
    }
}

@Preview
@Composable
private fun HistoryScreenPreview() {
    DermatoaiTheme {
        HistoryScreen()
    }
}