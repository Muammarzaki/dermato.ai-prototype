package com.github.dermatoai.ui.components

import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp

@Composable
fun ImageSelectionArea(imageUri: Uri?, onClick: () -> Unit) {
    Box(
        modifier = Modifier.Companion
            .fillMaxWidth()
            .height(200.dp)
            .clip(RoundedCornerShape(16.dp))
            .border(
                2.dp,
                MaterialTheme.colorScheme.outlineVariant,
                androidx.compose.foundation.shape.RoundedCornerShape(16.dp)
            )
            .background(MaterialTheme.colorScheme.surface)
            .clickable { onClick() },
        contentAlignment = Alignment.Companion.Center
    ) {
        if (imageUri != null) {
            Column(horizontalAlignment = Alignment.Companion.CenterHorizontally) {
                Icon(
                    Icons.Default.Image,
                    contentDescription = null,
                    modifier = Modifier.Companion.size(48.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Text("Image Selected!", color = MaterialTheme.colorScheme.primary)
            }
        } else {
            Column(horizontalAlignment = Alignment.Companion.CenterHorizontally) {
                Icon(
                    imageVector = Icons.Default.AddPhotoAlternate,
                    contentDescription = "Upload",
                    modifier = Modifier.Companion.size(48.dp),
                    tint = MaterialTheme.colorScheme.secondary
                )
                Spacer(modifier = Modifier.Companion.height(8.dp))
                Text("Tap to select image", color = MaterialTheme.colorScheme.secondary)
            }
        }
    }
}