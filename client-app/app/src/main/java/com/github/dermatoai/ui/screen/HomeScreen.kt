package com.github.dermatoai.ui.screen

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.History
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.paging.compose.LazyPagingItems
import androidx.paging.compose.collectAsLazyPagingItems
import com.github.dermatoai.domain.common.NetworkProtocol
import com.github.dermatoai.ui.components.HistoryItemCard
import com.github.dermatoai.ui.components.ImageSelectionArea
import com.github.dermatoai.ui.components.ProtocolSelector
import com.github.dermatoai.ui.dto.PredictionHistory
import com.github.dermatoai.ui.vm.AnalyzeVM
import com.github.dermatoai.ui.vm.DataVM


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    analyzeVM: AnalyzeVM = viewModel(),
    dataVM: DataVM = viewModel()
) {
    val uiState by analyzeVM.uiState.collectAsState()

    val historyItems = dataVM.historyPagingFlow.collectAsLazyPagingItems()

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> analyzeVM.onImageSelected(uri) }
    )

    Scaffold { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            ProtocolSelector(
                selected = uiState.selectedProtocol.name,
                onSelected = {
                    analyzeVM.onProtocolSelected(NetworkProtocol.valueOf(it))
                }
            )

            Spacer(modifier = Modifier.height(24.dp))

            ImageSelectionArea(
                imageUri = uiState.selectedImageUri,
                onClick = {
                    photoPickerLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                }
            )

            Spacer(modifier = Modifier.height(24.dp))

            Button(
                onClick = { analyzeVM.analyzeImage() },
                modifier = Modifier.fillMaxWidth(),
                enabled = uiState.selectedImageUri != null && !uiState.isLoading
            ) {
                if (uiState.isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp),
                        color = MaterialTheme.colorScheme.onPrimary,
                        strokeWidth = 2.dp
                    )
                } else {
                    Text("Analyze Image via ${uiState.selectedProtocol.name}")
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    Icons.Default.History,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Recent Predictions",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            HistoryList(historyItems)
        }
    }
}

@Composable
fun HistoryList(history: LazyPagingItems<PredictionHistory>) {
    LazyColumn(
        contentPadding = PaddingValues(bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(count = history.itemCount) { index ->
            val item = history[index]

            if (item != null) {
                HistoryItemCard(item)
            } else {
                Text("Loading item...")
            }
        }
    }
}


@Preview(showBackground = true)
@Composable
fun HomeScreenPreview() {
    MaterialTheme {
        HomeScreen()
    }
}