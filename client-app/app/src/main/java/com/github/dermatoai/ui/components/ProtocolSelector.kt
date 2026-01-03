package com.github.dermatoai.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun ProtocolSelector(selected: String, onSelected: (String) -> Unit) {
    Row(
        modifier = Modifier.Companion
            .fillMaxWidth()
            .clip(RoundedCornerShape(10))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(4.dp),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        val options = listOf("REST", "gRPC")
        options.forEach { option ->
            val isSelected = selected == option
            val containerColor =
                if (isSelected) MaterialTheme.colorScheme.primary else Color.Companion.Transparent
            val contentColor =
                if (isSelected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant

            Box(
                modifier = Modifier.Companion
                    .weight(1f)
                    .clip(androidx.compose.foundation.shape.RoundedCornerShape(10))
                    .background(containerColor)
                    .clickable { onSelected(option) }
                    .padding(vertical = 12.dp),
                contentAlignment = Alignment.Companion.Center
            ) {
                Text(
                    text = option,
                    color = contentColor,
                    fontWeight = if (isSelected) FontWeight.Companion.Bold else FontWeight.Companion.Normal
                )
            }
        }
    }
}