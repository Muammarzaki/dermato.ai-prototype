package com.github.dermatoai.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.github.dermatoai.ui.dto.PredictionHistory
import com.github.dermatoai.ui.theme.DermatoaiTheme

@Composable
fun PredictionResultDialog(
    result: PredictionHistory,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            Button(
                onClick = onDismiss,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Done")
            }
        },
        icon = {
            Icon(
                imageVector = Icons.Default.Analytics,
                contentDescription = null,
                modifier = Modifier.size(48.dp),
                tint = MaterialTheme.colorScheme.primary
            )
        },
        title = {
            Text(
                text = "Analysis Complete",
                style = MaterialTheme.typography.headlineSmall,
                textAlign = TextAlign.Center
            )
        },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.padding(vertical = 8.dp)
                ) {
                    Text(
                        text = "Prediction Result",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.secondary
                    )
                    Text(
                        text = result.prediction,
                        style = MaterialTheme.typography.headlineMedium,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.ExtraBold
                    )
                }

                HorizontalDivider()

                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .padding(12.dp)
                            .fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        MetricRow(
                            label = "Confidence",
                            value = result.getFormattedConfidence(),
                            icon = Icons.Default.CheckCircle,
                            iconTint = MaterialTheme.colorScheme.tertiary
                        )

                        MetricRow(
                            label = "Protocol",
                            value = result.method,
                            icon = if (result.method.equals(
                                    "GRPC",
                                    ignoreCase = true
                                )
                            ) Icons.Default.Bolt else Icons.Default.Cloud,
                            iconTint = MaterialTheme.colorScheme.secondary,
                            isChip = true,
                            chipColor = if (result.method.equals("GRPC", ignoreCase = true)) Color(
                                0xFF2196F3
                            ) else Color(0xFFE91E63)
                        )
                    }
                }
            }
        },
        containerColor = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(16.dp)
    )
}

@Composable
private fun MetricRow(
    label: String,
    value: String,
    icon: ImageVector,
    iconTint: Color,
    isChip: Boolean = false,
    chipColor: Color = Color.Unspecified
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = iconTint,
                modifier = Modifier.size(18.dp)
            )
            Text(
                text = " $label",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        if (isChip) {
            Surface(
                color = chipColor.copy(alpha = 0.1f),
                contentColor = chipColor,
                shape = RoundedCornerShape(8.dp),
                border = androidx.compose.foundation.BorderStroke(
                    1.dp,
                    chipColor.copy(alpha = 0.2f)
                )
            ) {
                Text(
                    text = value.uppercase(),
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }
        } else {
            Text(
                text = value,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
    }
}

@Preview
@Composable
private fun PredictionResultDialogPreview() {
    DermatoaiTheme {
        PredictionResultDialog(
            result = PredictionHistory(
                id = "1",
                imagePath = "",
                prediction = "Disease Name",
                confidence = 0.9,
                method = "REST",
                latency = 100,
                timestamp = System.currentTimeMillis(),
                isSuccess = true,
                rawId = 1L
            ),
            onDismiss = {}
        )
    }
}

@Preview
@Composable
private fun PredictionResultDialogPreview1() {
    DermatoaiTheme {
        PredictionResultDialog(
            result = PredictionHistory(
                id = "1",
                imagePath = "",
                prediction = "Disease Name",
                confidence = 0.9,
                method = "gRPC",
                latency = 100,
                timestamp = System.currentTimeMillis(),
                isSuccess = true,
                rawId = 1
            ),
            onDismiss = {}
        )
    }
}