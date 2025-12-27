package com.github.dermatoai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.github.dermatoai.screen.HomeScreen
import com.github.dermatoai.ui.theme.DermatoaiTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DermatoaiTheme {
                HomeScreen()
            }
        }
    }
}

