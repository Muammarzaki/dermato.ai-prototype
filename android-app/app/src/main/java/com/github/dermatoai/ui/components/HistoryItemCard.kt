package com.github.dermatoai.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.gestures.draggable
import androidx.compose.foundation.gestures.rememberDraggableState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import com.github.dermatoai.ui.dto.PredictionHistory
import com.github.dermatoai.ui.theme.DermatoaiTheme
import kotlin.math.roundToInt

@Composable
fun HistoryItemCard(
    item: PredictionHistory,
    onDelete: (Long) -> Unit
) {
    val haptic = LocalHapticFeedback.current
    val confidenceColor = when {
        item.confidence > 0.8 -> Color(0xFF2E7D32)
        item.confidence > 0.5 -> Color(0xFFF57F17)
        else -> Color(0xFFC62828)
    }
    var offsetX by remember { mutableStateOf(0f) }
    val threshold = 700f
    val animatedOffsetX by animateFloatAsState(
        targetValue = offsetX,
        label = "swipe"
    )
    val swipeProgress = (offsetX / threshold).coerceIn(0f, 2f)

    val isLatencyOutlier = item.latency > 1000
    Box {
        Card(
            modifier = Modifier
                .matchParentSize()
                .height(IntrinsicSize.Min),
            colors = CardDefaults.cardColors(containerColor = Color.Red),
            elevation = CardDefaults.cardElevation(0.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(end = 30.dp),
                contentAlignment = Alignment.CenterStart
            ) {
                Icon(
                    imageVector = Icons.Default.Delete,
                    contentDescription = "Delete",
                    tint = Color.White,
                    modifier = Modifier
                        .offset(x = 20.dp)
                        .size(50.dp)
                        .alpha(swipeProgress)
                )
            }
        }

        Card(
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            border = BorderStroke(1.dp, confidenceColor.copy(alpha = 0.5f)),
            modifier = Modifier
                .offset { IntOffset(animatedOffsetX.roundToInt(), 0) }
                .draggable(
                    orientation = Orientation.Horizontal,
                    state = rememberDraggableState { delta -> if (delta > 0) offsetX += delta },
                    onDragStopped = {
                        if (offsetX > threshold) {
                            haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                            onDelete(item.rawId)
                        } else {
                            offsetX = 0f
                        }
                    }
                )
                .fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(12.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Surface(
                        shape = RoundedCornerShape(4.dp),
                        color = if (item.method.contains("gRPC", true))
                            MaterialTheme.colorScheme.primaryContainer
                        else
                            MaterialTheme.colorScheme.secondaryContainer
                    ) {
                        Text(
                            text = item.method.uppercase(),
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Bold
                        )
                    }

                    Text(
                        text = item.getRelativeTime(),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = item.prediction,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(modifier = Modifier.height(4.dp))

                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            // Confidence Text
                            Text(
                                text = "Confidence: ${item.getFormattedConfidence()}",
                                style = MaterialTheme.typography.bodySmall,
                                color = confidenceColor,
                                fontWeight = FontWeight.SemiBold
                            )


                            Text(
                                text = item.getFormattedLatency(),
                                style = MaterialTheme.typography.bodySmall,
                                // Warna merah kalau latency outlier
                                color = if (isLatencyOutlier) Color.Red else MaterialTheme.colorScheme.onSurface
                            )

                            if (isLatencyOutlier || item.confidence < 0.5) {
                                Spacer(modifier = Modifier.width(4.dp))
                                Icon(
                                    imageVector = Icons.Default.Warning,
                                    contentDescription = "Outlier",
                                    tint = Color.Red,
                                    modifier = Modifier.size(14.dp)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Preview
@Composable
private fun HistoryItemCardPreview() {
    DermatoaiTheme {
        HistoryItemCard(
            item = PredictionHistory(
                id = "1",
                imagePath = "",
                prediction = "Disease Name",
                confidence = 0.9,
                method = "REST",
                latency = 100,
                timestamp = System.currentTimeMillis(),
                isSuccess = true,
                rawId = 1L,
            ),
            onDelete = {}
        )
    }
}