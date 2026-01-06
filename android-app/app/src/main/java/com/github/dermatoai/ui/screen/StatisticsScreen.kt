package com.github.dermatoai.ui.screen


import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.CloudQueue
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.github.dermatoai.ui.theme.DermatoaiTheme

@Composable
fun StatisticsScreen() {
    val restAvgLatency = 145
    val grpcAvgLatency = 38
    val totalScans = 1240
    val successRate = 98.5
    val grpcUsagePercent = 0.65f

    var startAnimation by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { startAnimation = true }

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
                    value = "$totalScans",
                    icon = Icons.Default.CloudQueue,
                    color = MaterialTheme.colorScheme.primary
                )
                SummaryCard(
                    modifier = Modifier.weight(1f),
                    title = "Success Rate",
                    value = "$successRate%",
                    icon = Icons.Default.CheckCircle,
                    color = MaterialTheme.colorScheme.tertiary
                )
            }

            // Latency Comparison Chart
            LatencyChartCard(
                restLatency = restAvgLatency,
                grpcLatency = grpcAvgLatency,
                animate = startAnimation
            )

            // Protocol Distribution
            ProtocolDistributionCard(
                grpcPercent = grpcUsagePercent,
                animate = startAnimation
            )
        }
    }
}

@Composable
fun SummaryCard(
    modifier: Modifier = Modifier,
    title: String,
    value: String,
    icon: ImageVector,
    color: Color
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(color.copy(alpha = 0.1f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(imageVector = icon, contentDescription = null, tint = color)
            }
            Text(
                text = value,
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.ExtraBold,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = title,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun LatencyChartCard(
    restLatency: Int,
    grpcLatency: Int,
    animate: Boolean
) {
    val maxLatency = maxOf(restLatency, grpcLatency) * 1.2f // Add buffer space
    val speedUpFactor = restLatency / grpcLatency

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Average Latency",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                Surface(
                    color = MaterialTheme.colorScheme.primaryContainer,
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        text = "${speedUpFactor}x Faster",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            MetricBar(
                label = "REST",
                value = restLatency,
                maxValue = maxLatency,
                color = Color(0xFFE91E63),
                icon = Icons.Default.Cloud,
                animate = animate
            )

            MetricBar(
                label = "gRPC",
                value = grpcLatency,
                maxValue = maxLatency,
                color = Color(0xFF2196F3),
                icon = Icons.Default.Bolt,
                animate = animate
            )

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            Text(
                text = "Lower is better. gRPC significantly outperforms REST in image transmission speed.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun MetricBar(
    label: String,
    value: Int,
    maxValue: Float,
    color: Color,
    icon: ImageVector,
    animate: Boolean
) {
    val animatedProgress by animateFloatAsState(
        targetValue = if (animate) value / maxValue else 0f,
        animationSpec = tween(durationMillis = 1000, easing = FastOutSlowInEasing),
        label = "barAnimation"
    )

    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth()
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(20.dp)
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
            modifier = Modifier.width(50.dp),
            fontWeight = FontWeight.SemiBold
        )

        Spacer(modifier = Modifier.width(12.dp))

        Box(modifier = Modifier.weight(1f)) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
            )
            Box(
                modifier = Modifier
                    .fillMaxWidth(animatedProgress)
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp))
                    .background(
                        Brush.horizontalGradient(
                            colors = listOf(color.copy(alpha = 0.7f), color)
                        )
                    )
            )
        }

        Spacer(modifier = Modifier.width(12.dp))

        Text(
            text = "${value}ms",
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface
        )
    }
}

@Composable
fun ProtocolDistributionCard(
    grpcPercent: Float,
    animate: Boolean
) {
    val animatedPercent by animateFloatAsState(
        targetValue = if (animate) grpcPercent else 0.5f,
        animationSpec = tween(durationMillis = 1000)
    )

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Protocol Usage",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                Icon(
                    imageVector = Icons.Default.Speed,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary
                )
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(12.dp)
                    .clip(RoundedCornerShape(6.dp))
            ) {
                Box(
                    modifier = Modifier
                        .weight(animatedPercent)
                        .fillMaxHeight()
                        .background(Color(0xFF2196F3))
                )
                Box(
                    modifier = Modifier
                        .weight(1f - animatedPercent)
                        .fillMaxHeight()
                        .background(Color(0xFFE91E63))
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                LegendItem(
                    label = "gRPC (${(animatedPercent * 100).toInt()}%)",
                    color = Color(0xFF2196F3)
                )
                LegendItem(
                    label = "REST (${100 - (animatedPercent * 100).toInt()}%)",
                    color = Color(0xFFE91E63)
                )
            }
        }
    }
}

@Composable
fun LegendItem(label: String, color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(color)
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
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