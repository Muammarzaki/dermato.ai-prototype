package com.github.dermatoai.ui.screen

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.github.dermatoai.ui.theme.DermatoaiTheme

@Composable
fun StatisticsScreen(
    // viewModel: DataVM = hiltViewModel() 
) {
    val restAvgLatency = 120
    val grpcAvgLatency = 45
    val totalScans = 150

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text("Performance Dashboard", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(24.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Total Predictions", style = MaterialTheme.typography.titleMedium)
                Text("$totalScans", style = MaterialTheme.typography.displayMedium)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text("Latency Comparison (Lower is Better)", fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(8.dp))

        // Visualisasi Bar Chart Sederhana
        MetricBar(label = "REST API", value = restAvgLatency, color = Color.Magenta)
        Spacer(modifier = Modifier.height(8.dp))
        MetricBar(label = "gRPC", value = grpcAvgLatency, color = Color.Blue)

        Spacer(modifier = Modifier.height(24.dp))

        Text("Insight:", fontWeight = FontWeight.SemiBold)
        Text("gRPC is approximately ${(restAvgLatency / grpcAvgLatency)}x faster than REST in current network conditions.")
    }
}

@Composable
fun MetricBar(label: String, value: Int, color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, modifier = Modifier.width(60.dp), fontWeight = FontWeight.Bold)

        // Bar
        Box(
            modifier = Modifier
                .height(24.dp)
                .width((value * 2).dp) // Scaling sederhana
                .clip(RoundedCornerShape(4.dp))
                .background(color)
        )

        Spacer(modifier = Modifier.width(8.dp))
        Text("$value ms")
    }
}

@Preview
@Composable
private fun StatisticsScreenPreview() {
    DermatoaiTheme {
        StatisticsScreen()
    }
}