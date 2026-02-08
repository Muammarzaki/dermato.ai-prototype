package com.github.dermatoai.ui.screen


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudQueue
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import com.github.dermatoai.ui.components.LatencyChartCard
import com.github.dermatoai.ui.components.ProtocolDistributionCard
import com.github.dermatoai.ui.components.ReliabilityCard
import com.github.dermatoai.ui.components.SummaryCard
import com.github.dermatoai.ui.theme.DermatoaiTheme
import com.github.dermatoai.ui.vm.StatisticsVM

@Composable
fun StatisticsScreen(
    viewModel: StatisticsVM = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    var startAnimation by remember { mutableStateOf(false) }
    LaunchedEffect(uiState.isLoading) {
        if (!uiState.isLoading) startAnimation = true
    }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Timeline,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(32.dp)
                )
                Spacer(modifier = Modifier.width(12.dp))
                Column {
                    Text(
                        text = "System Metrics",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Real-time performance stats",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                SummaryCard(
                    modifier = Modifier.weight(1f),
                    title = "Total Scans",
                    value = "${uiState.totalScans}",
                    icon = Icons.Default.CloudQueue,
                    color = MaterialTheme.colorScheme.primary
                )
                SummaryCard(
                    modifier = Modifier.weight(1f),
                    title = "Success Rate",
                    value = String.format("%.1f%%", uiState.overallSuccessRate),
                    icon = Icons.Default.CheckCircle,
                    color = if (uiState.overallSuccessRate > 90) Color(0xFF4CAF50) else Color(
                        0xFFFF9800
                    )
                )
            }

            // Latency Comparison Chart
            LatencyChartCard(
                restLatency = uiState.restStats.avgLatency,
                grpcLatency = uiState.grpcStats.avgLatency,
                animate = startAnimation
            )

            // Protocol Distribution
            ProtocolDistributionCard(
                grpcPercent = if (uiState.totalScans > 0) uiState.grpcStats.usagePercent else 0.5f,
                animate = startAnimation
            )

            ReliabilityCard(
                restStats = uiState.restStats,
                grpcStats = uiState.grpcStats
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun StatisticsScreenPreview() {
    DermatoaiTheme {
        StatisticsScreen()
    }
}