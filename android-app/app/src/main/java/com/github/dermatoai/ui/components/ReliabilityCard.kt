package com.github.dermatoai.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.github.dermatoai.ui.dto.ProtocolUiData

@Composable
fun ReliabilityCard(
    restStats: ProtocolUiData,
    grpcStats: ProtocolUiData
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(
                text = "Protocol Reliability",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            // Header Tabel Kecil
            Row(modifier = Modifier.fillMaxWidth()) {
                Text(text = "Protocol", modifier = Modifier.weight(1f), fontWeight = FontWeight.SemiBold)
                Text(text = "Success", modifier = Modifier.weight(1f), fontWeight = FontWeight.SemiBold, color = Color(0xFF4CAF50))
                Text(text = "Error", modifier = Modifier.weight(1f), fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.error)
            }
            
            HorizontalDivider()

            // Baris REST
            ReliabilityRow(
                name = "REST",
                success = restStats.successRate,
                error = restStats.errorRate
            )

            // Baris gRPC
            ReliabilityRow(
                name = "gRPC",
                success = grpcStats.successRate,
                error = grpcStats.errorRate
            )
        }
    }
}