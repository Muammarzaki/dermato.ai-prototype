package com.github.dermatoai.ui.screen

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Badge
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import com.github.dermatoai.ui.components.ImageSelectionArea
import com.github.dermatoai.ui.components.ProtocolSelector
import com.github.dermatoai.ui.dto.PredictionHistory
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

        // 1. Pilih Protokol
        ProtocolSelector(
            selected = uiState.selectedProtocol,
            onSelected = { viewModel.onProtocolSelected(it) }
        )

        Spacer(modifier = Modifier.height(24.dp))

        // 2. Area Gambar
        ImageSelectionArea(
            imageUri = uiState.selectedImageUri,
            onClick = {
                photoPickerLauncher.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            }
        )

        Spacer(modifier = Modifier.height(32.dp))

        // 3. Tombol Analyze
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

        // Error Message
        if (uiState.errorMessage != null) {
            Text(
                text = uiState.errorMessage!!,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(top = 16.dp)
            )
        }
    }

    // --- RESULT DIALOG / POPUP ---
    // Muncul otomatis ketika ada lastPredictionResult baru
    if (uiState.lastPredictionResult != null && !uiState.isLoading) {
        PredictionResultDialog(
            result = uiState.lastPredictionResult!!,
            onDismiss = { /* Reset state di VM jika perlu atau biarkan user menutupnya */ }
        )
    }
}

// Component Khusus untuk menampilkan Hasil & Metrik Teknis
@Composable
fun PredictionResultDialog(
    result: PredictionHistory, // Menggunakan model UI yang sudah dimapping
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Close") }
        },
        title = {
            Column {
                Text(text = "Analysis Result", style = MaterialTheme.typography.titleLarge)
                Text(
                    text = result.result,
                    style = MaterialTheme.typography.headlineMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold
                )
            }
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                HorizontalDivider()
                Text("Technical Metrics:", fontWeight = FontWeight.SemiBold)

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Protocol:")
                    Badge(containerColor = if (result.method == "GRPC") Color.Blue else Color.Magenta) {
                        Text(result.method, color = Color.White)
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Confidence:")
                    Text(result.confidence, fontWeight = FontWeight.Bold)
                }

                // Jika nanti Anda simpan latency di PredictionHistory, tampilkan disini
                // Row { Text("Latency:"); Text("${result.latency} ms") } 
            }
        }
    )
}