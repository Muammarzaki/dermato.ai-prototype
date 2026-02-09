package com.github.dermatoai.ui.screen


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudQueue
import androidx.compose.material.icons.filled.Insights
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
import androidx.compose.ui.text.style.TextAlign
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
    val total by viewModel.recordCount.collectAsState(initial = 0)

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background
    ) { padding ->

        if (total > 0) {
            StatisticsContent(
                modifier = Modifier.padding(padding),
                viewModel = viewModel
            )
        } else {
            EmptyStatisticsState(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
            )
        }
    }
}

@Composable
private fun StatisticsContent(
    modifier: Modifier = Modifier,
    viewModel: StatisticsVM
) {
    val uiState by viewModel.uiState.collectAsState()

    var startAnimation by remember { mutableStateOf(false) }
    LaunchedEffect(uiState.isLoading) {
        if (!uiState.isLoading) startAnimation = true
    }

    Column(
        modifier = modifier
            .fillMaxSize()
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

        LatencyChartCard(
            restLatency = uiState.restStats.avgLatencyMs,
            grpcLatency = uiState.grpcStats.avgLatencyMs,
            animate = startAnimation
        )

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

@Composable
private fun EmptyStatisticsState(
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = Icons.Default.Insights,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.6f),
            modifier = Modifier.size(72.dp)
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "No Statistics Available",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "You haven't performed any scans yet. Start a scan to see system performance statistics here.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )
    }
}

@Preview(showBackground = true)
@Composable
private fun StatisticsScreenPreview() {
    DermatoaiTheme {
        StatisticsScreen()
    }
}
