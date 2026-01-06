package com.github.dermatoai.ui.activity

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.github.dermatoai.ui.screen.MainScreen
import com.github.dermatoai.ui.theme.DermatoaiTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DermatoaiTheme {
                MainScreen()
            }
        }
    }
}