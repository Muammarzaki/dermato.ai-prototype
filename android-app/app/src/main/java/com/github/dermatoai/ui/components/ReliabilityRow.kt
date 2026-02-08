package com.github.dermatoai.ui.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight

@Composable
fun ReliabilityRow(name: String, success: Float, error: Float) {
    Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(text = name, modifier = Modifier.weight(1f), fontWeight = FontWeight.Medium)
        Text(
            text = String.format("%.1f%%", success), 
            modifier = Modifier.weight(1f), 
            color = Color(0xFF4CAF50) // Green
        )
        Text(
            text = String.format("%.1f%%", error), 
            modifier = Modifier.weight(1f), 
            color = MaterialTheme.colorScheme.error // Red
        )
    }
}