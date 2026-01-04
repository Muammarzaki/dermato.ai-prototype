package com.github.dermatoai.ui.screen

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import com.github.dermatoai.ui.components.ImageSelectionArea
import com.github.dermatoai.ui.components.PredictionResultDialog
import com.github.dermatoai.ui.components.ProtocolSelector
import com.github.dermatoai.ui.vm.AnalyzeVM

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AnalyzeScreen(
    viewModel: AnalyzeVM = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> viewModel.onImageSelected(uri) }
    )

    // Layout
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Dermato.AI Analysis", style = MaterialTheme.typography.headlineMedium)

        Spacer(modifier = Modifier.height(24.dp))

        ProtocolSelector(
            selected = uiState.selectedProtocol,
            onSelected = { viewModel.onProtocolSelected(it) }
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

        Spacer(modifier = Modifier.height(32.dp))

        Button(
            onClick = { viewModel.analyzeImage() },
            modifier = Modifier
                .fillMaxWidth()
                .height(50.dp),
            enabled = uiState.selectedImageUri != null && !uiState.isLoading
        ) {
            if (uiState.isLoading) {
                CircularProgressIndicator(color = Color.White)
            } else {
                Text("Start Analysis")
            }
        }

        if (uiState.errorMessage != null) {
            Text(
                text = uiState.errorMessage!!,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(top = 16.dp)
            )
        }
    }

    if (uiState.lastPredictionResult != null && !uiState.isLoading) {
        PredictionResultDialog(
            result = uiState.lastPredictionResult!!,
            onDismiss = { /* Reset state di VM jika perlu atau biarkan user menutupnya */ }
        )
    }
}